const pdfParse = require("pdf-parse");
const axios = require("axios");
const { Pinecone } = require('@pinecone-database/pinecone');
const { spawn } = require("child_process");

require('dotenv').config()

// Hugging Face Configuration
const HUGGING_FACE_API_KEY = process.env.HUGGING_FACE_API_KEY;
const HUGGING_FACE_MODEL = process.env.HUGGING_FACE_MODEL;

// Pinecone Configuration
const PINECONE_API_KEY = process.env.PINECONE_API_KEY;
const PINECONE_INDEX_NAME = process.env.PINECONE_INDEX_NAME;
const PINECONE_ENVIRONMENT = process.env.PINECONE_ENVIRONMENT;

const pinecone = new Pinecone({
    apiKey: PINECONE_API_KEY,
});

// Extract text from PDF
async function extractTextFromPDF(pdfBuffer) {
    const data = await pdfParse(pdfBuffer);
    return data.text;
}

// Split text into chunks
function chunkText(text, maxTokens = 300) {
    const sentences = text.match(/[^\.!\?]+[\.!\?]+/g) || [];
    const chunks = [];
    let currentChunk = "";

    for (const sentence of sentences) {
        if (currentChunk.length + sentence.length > maxTokens) {
            chunks.push(currentChunk);
            currentChunk = sentence;
        } else {
            currentChunk += sentence;
        }
    }

    if (currentChunk) {
        chunks.push(currentChunk);
    }

    return chunks;
}

// Function to get embeddings for a single text chunk
async function getEmbeddings(textChunk) {
    return new Promise((resolve, reject) => {
        const sanitizedText = JSON.stringify([textChunk]);
        const pythonProcess = spawn("python3", ["get_embeddings.py", sanitizedText]);

        pythonProcess.stdout.on("data", (data) => {
            try {
                const embeddings = JSON.parse(data.toString());
                resolve(embeddings[0]);
            } catch (error) {
                reject(`Error parsing output: ${error.message}`);
            }
        });

        pythonProcess.stderr.on("data", (data) => {
            reject(`Python error: ${data.toString()}`);
        });

        pythonProcess.on("close", (code) => {
            if (code !== 0) {
                reject(`Python script exited with code ${code}`);
            }
        });
    });
}

// Upsert vectors into Pinecone in batches
async function upsertToPinecone(embeddings, chunks, batchSize = 100) {
    const index = pinecone.Index(PINECONE_INDEX_NAME);
    const totalVectors = embeddings.length;
    console.log(`Total vectors to upsert: ${totalVectors}`);

    for (let i = 0; i < totalVectors; i += batchSize) {
        const batchEmbeddings = embeddings.slice(i, i + batchSize);
        const batchChunks = chunks.slice(i, i + batchSize);

        const vectorsArray = batchEmbeddings.map((embedding, j) => ({
            id: `chunk-${i + j}`,
            values: embedding,
            metadata: { text: batchChunks[j] },
        }));

        console.log(`Upserting batch ${i / batchSize + 1}: ${vectorsArray.length} vectors`);
        try {
            await index.upsert(vectorsArray);
            console.log(`Batch ${i / batchSize + 1} upserted successfully.`);
        } catch (error) {
            console.error(`Error during batch ${i / batchSize + 1} upsert: ${error.message}`);
        }
    }
}

// Main function
async function processPDF(filePath) {
    const pdfBuffer = require("fs").readFileSync(filePath);
    const text = await extractTextFromPDF(pdfBuffer);
    const chunks = chunkText(text);
    const embeddings = [];

    console.log(`Total chunks: ${chunks.length}`);

    // Process embeddings in smaller batches to reduce memory usage
    const embeddingBatchSize = 10;
    for (let i = 0; i < chunks.length; i += embeddingBatchSize) {
        const batchChunks = chunks.slice(i, i + embeddingBatchSize);
        console.log(`Generating embeddings for batch ${i / embeddingBatchSize + 1}`);

        const batchEmbeddings = await Promise.all(batchChunks.map(getEmbeddings));
        embeddings.push(...batchEmbeddings);

        console.log(`Batch ${i / embeddingBatchSize + 1} embeddings generated.`);
    }

    // Upsert embeddings to Pinecone in smaller batches
    await upsertToPinecone(embeddings, chunks, 100);
}

// Query Pinecone for similar texts
async function queryPinecone(queryEmbedding) {
    const index = pinecone.Index(PINECONE_INDEX_NAME);
    const queryRequest = {
        vector: queryEmbedding,
        topK: 5,
        includeMetadata: true,
    };

    const response = await index.query(queryRequest);
    return response.matches.map((match) => match.metadata.text);
}

// Function to query your database
async function queryDatabase(queryText) {
    const queryEmbedding = await getEmbeddings(queryText);
    const results = await queryPinecone(queryEmbedding);
    console.log("Query Results:", results);
}

// Usage example
(async () => {
    const pdfPath = "Tesla-Master-Plan.pdf";
    const indexes = await pinecone.listIndexes();
    console.log("Available indexes:", indexes);

    await processPDF(pdfPath);
})();

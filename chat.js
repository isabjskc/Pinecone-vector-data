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

// Function to generate the embedding for a query (question)
async function getQueryEmbedding(queryText) {
    return new Promise((resolve, reject) => {
        const sanitizedText = JSON.stringify([queryText]);
        const pythonProcess = spawn("python3", ["get_embeddings.py", sanitizedText]);

        pythonProcess.stdout.on("data", (data) => {
            try {
                const embeddings = JSON.parse(data.toString());
                resolve(embeddings[0]); // Return the embedding for the query
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

// Function to query Pinecone with the question embedding
async function queryPinecone(queryEmbedding) {
    const index = pinecone.Index(PINECONE_INDEX_NAME);
    const queryRequest = {
        vector: queryEmbedding,
        topK: 5, // Number of similar chunks to retrieve
        includeMetadata: true,
    };

    try {
        const response = await index.query(queryRequest);
        return response.matches.map((match) => match.metadata.text); // Return the text of the top matches
    } catch (error) {
        console.error("Error querying Pinecone:", error.message);
        return [];
    }
}

// Function to ask a question using the vector database and LLM
async function askQuestion(queryText) {
    // Step 1: Get the query embedding for the question
    const queryEmbedding = await getQueryEmbedding(queryText);

    // Step 2: Query Pinecone to retrieve similar document chunks
    const similarChunks = await queryPinecone(queryEmbedding);

    // Step 3: Combine the similar chunks to form the context for the LLM
    const context = similarChunks.join("\n");

    console.log('context: ', context);

    return 'Ran Successfully!';
}

// Example function to simulate querying the LLM (using OpenAI GPT in this case)
async function queryLLM(question, context) {
    // Here, we would call the OpenAI API (or another LLM) to generate an answer
    // using the question and context.
    // For example, using OpenAI's GPT-3 or GPT-4:

    const openai = require("openai"); // Example OpenAI client, make sure you have the OpenAI package installed

    const response = await openai.Completion.create({
        model: "gpt-4", // or any available model
        prompt: `${context}\n\nQuestion: ${question}\nAnswer:`,
        max_tokens: 150,
        temperature: 0.7,
    });

    return response.choices[0].text.trim();
}

// Example usage
(async () => {
    const query = "Who is the founder of Tesla, Company?";
    const answer = await askQuestion(query);
    console.log("Answer:", answer);
})();

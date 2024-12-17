require('dotenv').config();
const express = require('express');
const axios = require('axios');
const { PineconeClient } = require('@pinecone-database/pinecone');
const app = express();
const port = 3000;

// Variables de entorno
const GOOGLE_API_KEY = process.env.GOOGLE_API_KEY;
const PINECONE_API_KEY = process.env.PINECONE_API_KEY;
const PINECONE_REGION = process.env.PINECONE_REGION;
const HUGGINGFACE_API_KEY = process.env.HUGGINGFACE_API_KEY;

// Pinecone cliente
const pinecone = new PineconeClient();
pinecone.init({ apiKey: PINECONE_API_KEY, environment: PINECONE_REGION });

// Configurar el middleware para analizar el cuerpo de la solicitud
app.use(express.json());

// Inicializar contador de preguntas
let questionCount = 0;

// Función para obtener respuesta de Google Gemini
async function getGeminiResponse(prompt) {
    const url = `https://generative-ai.googleapis.com/v1beta2/models/gemini-1.0-pro:generateText?key=${GOOGLE_API_KEY}`;
    const payload = {
        prompt: prompt,
        temperature: 0.9,
        maxOutputTokens: 2048,
    };

    try {
        const response = await axios.post(url, payload);
        return response.data.text;
    } catch (error) {
        console.error('Error al obtener la respuesta de Gemini:', error);
        throw new Error('Error al procesar la solicitud de Gemini');
    }
}

// Función para realizar la búsqueda de similitud en Pinecone
async function getContextFromPinecone(query) {
    const index = pinecone.Index("chatbotmedver2024");

    const embeddings = await axios.post('https://api.huggingface.co/embed', {
        headers: { Authorization: `Bearer ${HUGGINGFACE_API_KEY}` },
        body: JSON.stringify({ inputs: query })
    });

    const vector = embeddings.data[0].embedding;
    
    const queryResult = await index.query({
        vector: vector,
        topK: 5,  // Devuelve los 5 documentos más similares
        includeMetadata: true
    });

    return queryResult.matches.map(match => match.metadata.text).join("\n");
}

// Ruta del chatbot
app.post('/chatbot', async (req, res) => {
    try {
        const { question } = req.body;

        if (!question) {
            return res.status(400).json({ message: "No se recibió ninguna pregunta" });
        }

        questionCount += 1;

        // Realizar búsqueda de similitud en Pinecone
        const contextText = await getContextFromPinecone(question);

        // Crear el prompt para el modelo generativo
        const fullPrompt = `
            Actúa como un asistente virtual amigable y servicial. Tu tarea es ayudar a responder preguntas utilizando la información más relevante disponible.

            Aquí tienes la información relevante que encontré en la base de datos:

            ${contextText}

            Por favor, responde a la siguiente pregunta de la manera más precisa y útil posible:

            Pregunta: ${question}

            Respuesta:
        `;

        // Obtener respuesta del modelo Gemini
        const responseText = await getGeminiResponse(fullPrompt);

        // Añadir un mensaje adicional para más ayuda
        const fullResponse = `${responseText}\n\n¿En qué más le puedo ayudar?`;

        // Registra la pregunta y la respuesta (aquí puedes almacenar en un archivo o base de datos)
        console.log(`Pregunta: ${question}`);
        console.log(`Respuesta: ${fullResponse}`);

        // Enviar la respuesta
        return res.status(200).json({
            data: fullResponse,
            message: "La petición se procesó correctamente"
        });
    } catch (error) {
        console.error('Error al procesar la petición:', error);
        return res.status(500).json({ message: "Se produjo un error al procesar la petición" });
    }
});

// Iniciar el servidor
app.listen(port, () => {
    console.log(`Servidor escuchando en http://localhost:${port}`);
});

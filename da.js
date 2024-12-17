import { GoogleGenerativeAI } from "@google/generative-ai";
import { Pinecone } from "@pinecone-database/pinecone";

const API_KEY = "AIzaSyC7HmJJsbwBl3mFCmdoQt_psFmsqDcO3JM";
const pineconeKey = "ca40ceca-1678-4efc-ac07-146e28c1bfb7";

// Crear instancias
const genAI = new GoogleGenerativeAI(API_KEY);
const pc = new Pinecone({ apiKey: pineconeKey });
const indexName = "nuevomedvet";
const question = "tiene horarios?";  // Pregunta con errores ortográficos

async function run() {
  try {
    console.log(`Consultando el índice "${indexName}"...`);
    const index = pc.Index(indexName);

    // Vector de prueba
    const testVector = new Array(384).fill(0.5); // Este vector de prueba es un ejemplo; normalmente, el vector sería calculado basado en la consulta

    // Consultar Pinecone
    const result = await index.query({
      vector: testVector,
      topK: 1,
      includeMetadata: true,
    });

    if (result.matches?.length) {
      const context = result.matches
        .map((match) => match.metadata?.text)
        .join("\n");

      console.log("Contexto relevante encontrado:", context);

      // Limpiar el texto para eliminar marcas como '###' y saltos de línea innecesarios
      const cleanedContext = context.replace(/###/g, '').replace(/\n/g, ' ');

      // Generar respuesta con Gemini
      const prompt = `
        Eres un asistente virtual diseñado para ayudar a usuarios con consultas. A veces, los usuarios cometen errores ortográficos, pero tú deberías ser capaz de comprender su intención y proporcionar respuestas correctas.

        Aquí está la información relevante encontrada en la base de datos:
        ${cleanedContext}

        Pregunta del usuario: ${question}

        Responde clara y específicamente a la pregunta del usuario utilizando la información proporcionada. Si la pregunta tiene errores ortográficos, aún así, responde de manera precisa.
      `;

      // Crear la conversación con Gemini y enviar el prompt
      const chat = genAI.getGenerativeModel({ model: "gemini-pro" }).startChat();
      const response = await chat.sendMessage(prompt);

      // Acceder correctamente al texto de la respuesta generada
      const generatedText = response?.text || "No se pudo generar una respuesta.";

      console.log("Respuesta generada:", generatedText);
    } else {
      console.log("No se encontró información relevante en la base de datos.");
    }
  } catch (error) {
    console.error("Error procesando la solicitud:", error.message);
  }
}

run();

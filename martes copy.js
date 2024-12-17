import { Pinecone } from '@pinecone-database/pinecone';
import { pipeline, layer_norm } from '@xenova/transformers';
import { GoogleGenerativeAI } from "@google/generative-ai";

// Inicializa Pinecone y el modelo de embeddings
const pc = new Pinecone({ apiKey: 'ca40ceca-1678-4efc-ac07-146e28c1bfb7' });

async function init() {
  // Aquí se inicializa el extractor con 'await' dentro de una función 'async'
  const extractor = await pipeline('feature-extraction', 'nomic-ai/nomic-embed-text-v1.5');

  // Crear una instancia de GoogleGenerativeAI con la API Key
  const API_KEY = "AIzaSyC7HmJJsbwBl3mFCmdoQt_psFmsqDcO3JM";
  const genAI = new GoogleGenerativeAI(API_KEY);

  // Obtener el modelo generativo de Gemini
  const model = genAI.getGenerativeModel({
    model: "gemini-1.0-pro",
  });

  // Configuración para la generación de respuestas
  const generationConfig = {
    temperature: 1,            // Controla la aleatoriedad de la respuesta
    topP: 0.95,               // Controla el recorte de la probabilidad acumulada
    topK: 40,                 // Controla cuántos posibles tokens usar para la generación
    maxOutputTokens: 8192,    // Número máximo de tokens de salida
    responseMimeType: "text/plain", // Tipo MIME de la respuesta
  };

  // Función para realizar la consulta a Pinecone y generar respuesta con Gemini
  async function consultarYGenerarRespuesta(query) {
    const indexName = "nuevomedvet"; // Cambia aquí por el nombre del índice que deseas verificar
    const indices = await pc.listIndexes();

    console.log('Índices disponibles:', indices); // Muestra los índices disponibles para depuración

    // Verifica si el índice existe
    if (Array.isArray(indices.indexes) && indices.indexes.some(index => index.name === indexName)) {
      console.log(`El índice "${indexName}" ya existe.`);

      // Extrae embeddings para la consulta
      let embeddings = await extractor([query], { pooling: 'mean' });

      // Normaliza y ajusta dimensiones
      const matryoshka_dim = 384; // Cambia esto a la dimensión correcta del índice
      embeddings = layer_norm(embeddings, [embeddings.dims[1]])
        .slice(null, [0, matryoshka_dim])
        .normalize(2, -1)
        .tolist();

      // Realiza la consulta de similitud
      const index = pc.Index(indexName);
      const resultado = await index.query({
        vector: embeddings[0],
        topK: 5,  // Número de resultados deseados
        includeMetadata: true
      });

      console.log("Resultados de la búsqueda en Pinecone:", JSON.stringify(resultado, null, 2));
      
      // Aquí es donde defines la pregunta
      const pregunta = query;  // Usamos la misma pregunta que pasaste

      // Extrae los fragmentos relevantes de Pinecone
      const contextoPinecone = resultado.matches.map(match => match.metadata.text).join("\n");
      
      // Estructura de prompt para la pregunta
      const prompt = `
      Aquí tienes la información relevante basada en Pinecone:
      
      ${contextoPinecone}
      
      Por favor, responde a la siguiente pregunta de manera clara y detallada, utilizando el contexto proporcionado:
      
      Pregunta: ${pregunta}
      
      Respuesta:
      `;

      // Genera una respuesta usando Gemini IA
      const chatSession = model.startChat({
        generationConfig,
        history: [], // No historial en este caso
      });

      const result = await chatSession.sendMessage(prompt);
      
      // Solo muestra la respuesta generada por Gemini
      console.log("Respuesta generada por Gemini:", result.response.text());
    }
  }

  // Llama a la función con tu pregunta específica
  await consultarYGenerarRespuesta("¿tiene chips??");
}

init().catch(console.error);

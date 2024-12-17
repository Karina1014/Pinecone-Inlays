import { GoogleGenerativeAI } from "@google/generative-ai";
import { Pinecone } from '@pinecone-database/pinecone';
import { pipeline, layer_norm } from '@xenova/transformers';

// Claves de API
const geminiApiKey = "AIzaSyC7HmJJsbwBl3mFCmdoQt_psFmsqDcO3JM";
const pineconeApiKey = "ca40ceca-1678-4efc-ac07-146e28c1bfb7";

// Inicializa Google Gemini
const genAI = new GoogleGenerativeAI(geminiApiKey);
const model = genAI.getGenerativeModel({
  model: "gemini-1.0-pro",
});

// Inicializa Pinecone y el extractor de embeddings
const pc = new Pinecone({ apiKey: pineconeApiKey });
const extractor = await pipeline('feature-extraction', 'nomic-ai/nomic-embed-text-v1.5');

// Configuración de Gemini
const generationConfig = {
  temperature: 1,            // Controla la aleatoriedad de la respuesta
  topP: 0.95,               // Controla el recorte de la probabilidad acumulada
  topK: 40,                 // Controla cuántos posibles tokens usar para la generación
  maxOutputTokens: 8192,    // Número máximo de tokens de salida
  responseMimeType: "text/plain", // Tipo MIME de la respuesta
};

// Función para consultar el índice de Pinecone
async function verificarYConsultarIndice(textoGenerado) {
  const indexName = "chatbotmedver2024"; // Nombre del índice
  const indices = await pc.listIndexes();

  // Verifica si el índice existe
  if (Array.isArray(indices.indexes) && indices.indexes.some(index => index.name === indexName)) {
    console.log(`El índice "${indexName}" ya existe.`);
    
    // Extrae embeddings para la consulta
    let embeddings = await extractor([textoGenerado], { pooling: 'mean' });
    
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
      topK: 1,  // Número de resultados deseados
      includeMetadata: true
    });

    // Muestra solo el campo "text" de los resultados
    if (resultado.matches && resultado.matches.length > 0) {
      console.log("Texto más similar encontrado:");
      resultado.matches.forEach((match, index) => {
        console.log(`Resultado ${index + 1}: ${match.metadata?.text || 'No disponible'}`);
      });
    } else {
      console.log("No se encontraron resultados similares.");
    }

  } else {
    console.log(`El índice "${indexName}" no existe.`);
  }
}

// Función principal para generar la respuesta de Gemini y consultar Pinecone
async function run() {
  try {
    // Iniciar la sesión de chat con Gemini
    const chatSession = model.startChat({
      generationConfig,
      history: [], // No historial en este caso
    });

    // Enviar el mensaje "¿Qué hace el presidente Noboa?"
    const result = await chatSession.sendMessage("¿Qué hace el presidente Noboa?");
    
    // Mostrar la respuesta generada por Gemini
    const textoGenerado = result.response.text();
    console.log("Texto generado por Gemini:", textoGenerado);
    
    // Llamar a la función para verificar y consultar Pinecone con el texto generado
    await verificarYConsultarIndice(textoGenerado);
    
  } catch (error) {
    console.error("Hubo un error al generar respuesta o al consultar Pinecone:", error);
  }
}

// Ejecutar la función
run();

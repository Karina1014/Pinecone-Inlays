import { Pinecone } from '@pinecone-database/pinecone';
import { pipeline } from '@xenova/transformers';

// Inicializa Pinecone y el modelo de embeddings
const pc = new Pinecone({ apiKey: 'ca40ceca-1678-4efc-ac07-146e28c1bfb7' });
const extractor = await pipeline('feature-extraction', 'nomic-ai/nomic-embed-text-v1.5');

async function buscarSimilitud() {
    const indexName = "chatbotmedver2024"; // Nombre del índice de Pinecone
    const pregunta = "¿Tiene chip?"; // Pregunta a consultar

    // Extrae los embeddings de la pregunta
    let embeddings = await extractor([pregunta], { pooling: 'mean' });

    // Verifica si los embeddings son un arreglo y extrae el vector adecuado
    let vector = embeddings[0]; // Embedding de la primera pregunta (debería ser un arreglo)

    if (!Array.isArray(vector)) {
        console.error("Error: los embeddings no son un arreglo");
        return;
    }

    // Realiza la consulta de similitud en Pinecone
    const index = pc.Index(indexName);
    const resultado = await index.query({
        vector: vector,  // Vector de la pregunta
        topK: 1,  // Número de resultados deseados
        includeMetadata: true
    });

    // Imprime los resultados
    console.log("Tu pregunta: ", pregunta);
    if (resultado.matches && resultado.matches.length > 0) {
        console.log("Resultado >>:", resultado.matches[0].metadata ? resultado.matches[0].metadata.text : "No hay texto disponible");
    } else {
        console.log("No se encontraron resultados similares.");
    }
}

// Llamar a la función de búsqueda
buscarSimilitud().catch(console.error);

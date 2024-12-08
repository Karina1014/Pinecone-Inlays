import { pipeline } from '@xenova/transformers';  // Importa el pipeline de Xenova
import Pinecone from '@pinecone-database/pinecone';  // Importa el cliente de Pinecone

// Inicializa el cliente de Pinecone
const pc = new Pinecone();

async function obtenerVector(texto) {
    // Inicializa el pipeline de feature-extraction para generar embeddings
    const extractor = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2');
    
    // Calcula los embeddings para la consulta
    const output = await extractor(texto, { pooling: 'mean', normalize: true });
    
    // El output devuelve un tensor; accede a sus datos (array de floats)
    return output.data;
}

async function realizarConsulta() {
    const indexName = "chatbotmedver2024";  // El nombre de tu índice en Pinecone
    const pregunta = "¿Tiene chip?";  // La consulta que quieres hacer
    
    // Obtén el vector de embeddings para la consulta
    const queryVector = await obtenerVector(pregunta);
    console.log("Vector de consulta:", queryVector);

    // Inicializa el cliente de Pinecone
    await pc.init({
        apiKey: 'ca40ceca-1678-4efc-ac07-146e28c1bfb7',  // Tu clave API de Pinecone
        environment: 'us-east-1'  // La región de tu índice
    });

    try {
        // Realiza la consulta al índice de Pinecone con el vector generado
        const queryResult = await pc.query({
            indexName: indexName,
            queryRequest: {
                vector: queryVector,  // El vector de la consulta
                topK: 5,              // Número de resultados a devolver
                includeValues: true,  // Incluir valores de los vectores en la respuesta
                includeMetadata: true // Incluir metadatos en la respuesta
            }
        });

        // Muestra los resultados de la consulta
        console.log('Resultados de la consulta:', queryResult);
    } catch (err) {
        console.error('Error en la consulta:', err);
    }
}

// Ejecuta la consulta
realizarConsulta().catch(console.error);

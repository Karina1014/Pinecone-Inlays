import { Pinecone } from '@pinecone-database/pinecone';
import { pipeline, layer_norm } from '@xenova/transformers';

// Inicializa Pinecone y el modelo de embeddings
const pc = new Pinecone({ apiKey: 'ca40ceca-1678-4efc-ac07-146e28c1bfb7' });
const extractor = await pipeline('feature-extraction', 'nomic-ai/nomic-embed-text-v1.5');

async function verificarYConsultarIndice() {
    const indexName = "chatbotmedver2024"; // Cambia aquí por el nombre del índice que deseas verificar
    const indices = await pc.listIndexes();

    console.log('Índices disponibles:', indices); // Muestra los índices disponibles para depuración

    // Verifica si el índice existe
    if (Array.isArray(indices.indexes) && indices.indexes.some(index => index.name === indexName)) {
        console.log(`El índice "${indexName}" ya existe.`);
        
        // Extrae embeddings para la consulta
        const consulta = ['search_query: ¿Cuáles son los horarios de atención?'];
        let embeddings = await extractor(consulta, { pooling: 'mean' });
        
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
        
        // Verifica y muestra los metadatos para identificar el campo de respuesta
        if (resultado.matches && resultado.matches.length > 0) {
            resultado.matches.forEach((match, index) => {
                console.log(`Resultado ${index + 1}:`, match.metadata); // Muestra todos los metadatos
            });

            // Intenta extraer la respuesta de los campos conocidos
            const respuestas = resultado.matches.map(match => match.metadata.respuesta || match.metadata.content);
            console.log("Respuesta a la consulta:", respuestas[0] || "Campo de respuesta no encontrado en los metadatos.");
        } else {
            console.log("No se encontraron coincidencias en la consulta.");
        }
    } else {
        console.log(`El índice "${indexName}" no existe.`);
    }
}

// Llama a la función para verificar y consultar el índice
verificarYConsultarIndice().catch(console.error);

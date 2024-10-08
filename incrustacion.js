import { Pinecone } from '@pinecone-database/pinecone';

// Inicializa un cliente de Pinecone
const pc = new Pinecone({ apiKey: 'ca40ceca-1678-4efc-ac07-146e28c1bfb7' });

async function crearIndice() {
    const indexName = "example-index-en-js";

    // Verifica si el índice ya existe
    const indices = await pc.listIndexes();
    console.log('Índices disponibles:', indices); // Agrega esta línea para depuración

    // Asegúrate de que indices sea un array
    const indicesArray = Array.isArray(indices) ? indices : indices.indexes || []; // Ajusta según la estructura real

    if (indicesArray.includes(indexName)) {
        console.log(`El índice "${indexName}" ya existe.`);
        return;
    }

    // Crea un índice serverless
    await pc.createIndex({
        name: indexName,
        dimension: 300, 
        metric: 'cosine',
        spec: {
            serverless: {
                cloud: 'aws',
                region: 'us-east-1'
            }
        }
    });

    console.log(`Índice "${indexName}" creado exitosamente.`);
}

// Llama a la función para crear el índice
crearIndice().catch(console.error);

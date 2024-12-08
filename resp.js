import { Pinecone } from '@pinecone-database/pinecone';

async function listarIndices() {
    const pc = new Pinecone({ apiKey: 'ca40ceca-1678-4efc-ac07-146e28c1bfb7' });
    
    // Lista todos los índices disponibles
    const indices = await pc.listIndexes();
    
    console.log('Índices disponibles:', indices);
}

// Llamar a la función para listar los índices
listarIndices().catch(console.error);

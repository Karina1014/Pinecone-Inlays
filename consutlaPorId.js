import { Pinecone } from '@pinecone-database/pinecone';

const pc = new Pinecone({ apiKey: 'ca40ceca-1678-4efc-ac07-146e28c1bfb7' });
const index = pc.index("chatbotmedver2024");

async function queryPinecone() {
    try {
        const queryResponse = await index.namespace('chatbotmedver2024').query({
            id: 'c',
            topK: 3,
            includeValues: true
        });
        console.log("Consulta exitosa:", queryResponse);
    } catch (error) {
        console.error("Error en la consulta:", error);
    }
}

import { Pinecone } from '@pinecone-database/pinecone';
import { HfInference } from '@huggingface/inference';

// Configuración de la API de Pinecone y Hugging Face
const pc = new Pinecone({ apiKey: 'ca40ceca-1678-4efc-ac07-146e28c1bfb7' });
const index = pc.index("chatbotmedver2024");
const hf = new HfInference('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'); // Reemplaza con tu token de acceso real

async function queryPinecone() {
    try {
        // Generar el embedding para la pregunta
        const inputText = "TIENE HORARIO";
        const embeddingResponse = await hf.embeddings({
            model: 'sentence-transformers/all-MiniLM-L6-v2',
            inputs: [inputText] // Debes pasar un array
        });

        const embeddingVector = embeddingResponse[0].embedding; // Acceder al embedding

        // Realizar la consulta en Pinecone
        const queryResponse = await index.namespace('chatbotmedver2024').query({
            vector: embeddingVector,
            topK: 3,
            includeValues: true
        });

        console.log("Consulta exitosa:", queryResponse);
    } catch (error) {
        console.error("Error en la consulta:", error);
    }
}

queryPinecone();

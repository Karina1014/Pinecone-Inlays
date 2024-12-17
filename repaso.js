import { Pinecone } from '@pinecone-database/pinecone';

const pineconeClient = new PineconeClient();

await pineconeClient.init({
  apiKey: 'ca40ceca-1678-4efc-ac07-146e28c1bfb7', // Sustituye con tu clave de API
  environment: 'us-east-1' // Asegúrate de que coincida con el entorno de tu Pinecone
});

const indices = await pineconeClient.listIndexes();

if (indices.includes("nuevomedvet")) {
    console.log("El índice 'nuevomedvet' ya existe.");
} else {
    console.log("El índice 'nuevomedvet' no existe.");
}

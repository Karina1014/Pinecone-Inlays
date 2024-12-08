import {
    GoogleGenerativeAI,
    HarmCategory,
    HarmBlockThreshold,
  } from "@google/generative-ai";
  
  const MODEL_NAME = "gemini-1.0-pro"; // Asegúrate de que este modelo soporte embeddings
  const API_KEY = "AIzaSyAWnm3H-6FSgzkiQ57bYxVEXZF2GW4CMHo"; // Asegúrate de que esta clave esté protegida
  
  async function obtenerEmbeddings(texto) {
    const genAI = new GoogleGenerativeAI(API_KEY);
    const model = genAI.getGenerativeModel({ model: MODEL_NAME });
  
    try {
        // Realiza la solicitud para obtener embeddings
        const embeddingsResponse = await model.getEmbeddings({ 
            inputs: [texto] // Envía el texto como un array
        });
  
        // Procesa y devuelve los embeddings
        const embeddings = embeddingsResponse.data; // Asegúrate de acceder correctamente a los datos
        console.log('Embeddings:', embeddings); // Imprime los embeddings
        return embeddings;
    } catch (error) {
        console.error("Error al obtener embeddings:", error);
        return null; // Manejo de errores
    }
  }
  
  // Ejemplo de uso
  const textoEjemplo = "Este es un ejemplo de texto para obtener embeddings.";
  obtenerEmbeddings(textoEjemplo);
  
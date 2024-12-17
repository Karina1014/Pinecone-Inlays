import {
    GoogleGenerativeAI,
    HarmCategory,
    HarmBlockThreshold,
  } from "@google/generative-ai";
  

const API_KEY = "AIzaSyC7HmJJsbwBl3mFCmdoQt_psFmsqDcO3JM";
  
// Crear una instancia de GoogleGenerativeAI con la API Key
const genAI = new GoogleGenerativeAI(API_KEY);

// Obtener el modelo generativo de Gemini
const model = genAI.getGenerativeModel({
  model: "gemini-1.0-pro",
});

// Configuración para la generación de respuestas
const generationConfig = {
  temperature: 1,            // Controla la aleatoriedad de la respuesta
  topP: 0.95,               // Controla el recorte de la probabilidad acumulada
  topK: 40,                 // Controla cuántos posibles tokens usar para la generación
  maxOutputTokens: 8192,    // Número máximo de tokens de salida
  responseMimeType: "text/plain", // Tipo MIME de la respuesta
};

async function run() {
  try {
    // Iniciar la sesión de chat con el modelo y configuración
    const chatSession = model.startChat({
      generationConfig,
      history: [], // No historial en este caso
    });

    // Enviar el mensaje "pais más pequeño"
    const result = await chatSession.sendMessage("¿Cómo cuidar a un perro Beagle?");
    
    // Mostrar la respuesta
    console.log(result.response.text());
  } catch (error) {
    console.error("Error al generar respuesta:", error);
  }
}

// Ejecutar la función
run();

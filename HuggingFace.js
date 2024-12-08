// Importar la clase HfInference desde el paquete de Hugging Face
import { HfInference } from '@huggingface/inference';

// Inicializar HfInference con tu token de acceso
const hf = new HfInference('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'); // Reemplaza con tu token real

// Función para realizar la consulta de question answering sobre la tabla
async function tableQuestionAnswering() {
  try {
    // Realizar la consulta
    const response = await hf.tableQuestionAnswering({
      model: 'google/tapas-large-finetuned-wtq', // Usar el modelo de TAPAS
      inputs: {
        query: '¿De qué color es la manzana?', // Nueva pregunta
        table: {
          Fruta: ['Manzana', 'Banana', 'Naranja'], // Nombres de frutas
          Color: ['Rojo', 'Amarillo', 'Naranja'], // Colores correspondientes
          Sabor: ['Dulce', 'Dulce', 'Ácido'], // Sabores correspondientes
        }
      }
    });

    // Mostrar la respuesta en consola
    console.log('Respuesta:', response);
  } catch (error) {
    // Manejo de errores
    console.error('Error en la consulta:', error);
  }
}

// Ejecutar la función
tableQuestionAnswering();

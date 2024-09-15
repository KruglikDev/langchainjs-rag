import { Ollama } from "@langchain/ollama";

class PdfQA {

    constructor({ model }) {
        this.model = model;
    }

    init(){
        this.initChatModel();
        return this;
    }

    // Make this method async in order to use the await keyword inside the function body:
    async initChatModel(){
        console.log("Loading model...");
        this.llm = new Ollama({ model: this.model });
        // Use the invoke() method available to ask a question to the LLM
        const response = await this.llm.invoke("Who was the first black man in baseball league?");
        console.log(response);
    }

}

const pdfQa = new PdfQA({ model: "llama3" }).init();
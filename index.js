import { Ollama, OllamaEmbeddings } from "@langchain/ollama";
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import path from "node:path";
import { CharacterTextSplitter } from "@langchain/textsplitters";
import { MemoryVectorStore } from 'langchain/vectorstores/memory';
import { createRetrievalChain } from "langchain/chains/retrieval";
import { createStuffDocumentsChain } from "langchain/chains/combine_documents";
import { ChatPromptTemplate, MessagesPlaceholder } from "@langchain/core/prompts";
import { HumanMessage, AIMessage } from "@langchain/core/messages";

const pdfDocumentPath = "./materials/pycharm-documentation-mini.pdf";
const selectEmbedding = new OllamaEmbeddings({model: 'all-minilm:latest'});

const questionAnsweringPrompt = ChatPromptTemplate.fromMessages([
    [
        "system",
        "You are an expert in AI topics. You are provided multiple context items that are related to the prompt you have to answer. Use the following pieces of context to answer the question at the end.\n\n{context}",
    ],
    new MessagesPlaceholder("chat_history"),
    ["human", "{input}"],
]);

function initChatModel(model) {
    console.log("Loading model...");
    return new Ollama({ model });
}

async function loadDocuments(pdfDocument) {
    console.log("Loading PDFs...");
    const pdfLoader = new PDFLoader(path.join(import.meta.dirname, pdfDocument));
    return await pdfLoader.load();
}

async function initPdfQA({ model, pdfDocument, chunkSize, chunkOverlap, kDocuments, searchType }) {
    const llm = initChatModel(model);
    const documents = await loadDocuments(pdfDocument);
    const texts = await splitDocuments({documents, chunkSize, chunkOverlap});
    const db = await createVectorStore(texts);
    const retriever = createRetriever({db, kDocuments, searchType})
    const chain = await createChain({llm, retriever});

    const chatHistory = [];

    return { llm, documents, texts, db, retriever, chain, chatHistory };
}

async function splitDocuments({documents, chunkSize, chunkOverlap}) {
    console.log("Splitting PDFs...");
    const splitter = new CharacterTextSplitter({
        separator: " ",
        chunkSize,
        chunkOverlap,
    });
    return await splitter.splitDocuments(documents);
}

async function createVectorStore(texts) {
    console.log('Creating document embeddings...');
    return await MemoryVectorStore.fromDocuments(texts, selectEmbedding);
}

function createRetriever({db, searchType, kDocuments}){
    console.log("Initialize vector store retriever...");
    return db.asRetriever({
        k: kDocuments,
        searchType
    });
}

async function createChain({llm, retriever}) {
    console.log('Creating retrieval QA chain...');
    const combineDocsChain = await createStuffDocumentsChain({
        llm: llm,
        prompt: questionAnsweringPrompt,
    });

    const retrievalChain = createRetrievalChain({
        retriever,
        combineDocsChain,
    });

    return retrievalChain;
}

// Function to invoke the chain with chat history and update it
async function askQuestion(pdfQa, question) {
    const { chain, chatHistory } = pdfQa;

    // Invoke the chain with the current question and chat history
    const result = await chain.invoke({
        input: question,
        chat_history: chatHistory // Pass in the chat history
    });

    // Store the new question and answer in the chat history using HumanMessage and AIMessage
    chatHistory.push(new HumanMessage(question));
    chatHistory.push(new AIMessage(result.answer));

    return result.answer;
}

const pdfQa = await initPdfQA({
    model: "llama3",
    pdfDocument: pdfDocumentPath,
    chunkSize: 1000,
    chunkOverlap: 0,
    searchType: 'similarity',
    kDocuments: 5
});

// Example of asking questions and updating the chat history
const firstQuestion = "What is the capital of UK?";
const firstAnswer = await askQuestion(pdfQa, firstQuestion);
console.log(firstAnswer);

const followUpQuestion = "What is the population of the capital?";
const secondAnswer = await askQuestion(pdfQa, followUpQuestion);
console.log(secondAnswer);


// Поиск по эмбеддингам, ищет не дословно, а по релевантности
// const similaritySearchResults = await pdfQa.db.similaritySearch("File type associations", 10);
// for (const doc of similaritySearchResults) {
//     console.log(JSON.stringify(doc.metadata.loc, null));
// }

// Поиск с показателем релевантности, можно например возвращать только те, у кого показатель выше 0.5
// const similaritySearchWithScore = await pdfQa.db.similaritySearchVectorWithScore(await selectEmbedding.embedQuery('File type associations'), 10);
// for (const [doc, score] of similaritySearchWithScore) {
//     console.log(`* [SIM=${score.toFixed(3)}] [Page number: ${doc.metadata.loc.pageNumber}]`)
// }
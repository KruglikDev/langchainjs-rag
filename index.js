import {Ollama, OllamaEmbeddings} from "@langchain/ollama";
import {PDFLoader} from "@langchain/community/document_loaders/fs/pdf";
import path from "node:path";
import {CharacterTextSplitter} from "@langchain/textsplitters";
import {MemoryVectorStore} from 'langchain/vectorstores/memory';
import { createRetrievalChain } from "langchain/chains/retrieval";
import { createStuffDocumentsChain } from "langchain/chains/combine_documents";
import {ChatPromptTemplate} from "@langchain/core/prompts";

const pdfDocumentPath = "./materials/pycharm-documentation-mini.pdf";
const selectEmbedding = new OllamaEmbeddings({model: 'all-minilm:latest'})

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
    return { llm, documents, texts, db, retriever, chain };
}

// Разбивает документ на небольшие чанки для ,
// т.к. у llm есть ограничение на количество обрабатываемых токенов за раз
async function splitDocuments({documents, chunkSize, chunkOverlap}) {
    console.log("Splitting PDFs...");
    const splitter = new CharacterTextSplitter({
        separator: " ",
        chunkSize,
        chunkOverlap,
    })
    return await splitter.splitDocuments(documents)
}

// Векторное хранилище, где чанки сохраняются как эмбеддинги
async function createVectorStore(texts) {
    console.log('Creating document embeddings...');

    return await MemoryVectorStore.fromDocuments(texts, selectEmbedding);
}

// Функция: createRetriever создает объект, который позволяет искать по векторному хранилищу документов.
// Этот объект отвечает за извлечение наиболее релевантных документов из хранилища на основе заданного запроса.
// Он использует метод поиска (например, по схожести) для нахождения документов, которые лучше всего соответствуют запросу пользователя.
function createRetriever({db, searchType, kDocuments}){
    console.log("Initialize vector store retriever...");
    return db.asRetriever({
        k: kDocuments,
        searchType
    });
}

async function createChain({llm, retriever}) {
    console.log('Creating retrieval QA chain...');

    const questionAnsweringPrompt = ChatPromptTemplate.fromMessages([
        [
            "system",
            "You are an expert in AI topics. You are provided multiple context items that are related to the prompt you have to answer. Use the following pieces of context to answer the question at the end.\n\n{context}",
        ],
        ["human", "{input}"],
    ]);

    // Объединяет и обрабатывает документы, чтобы подготовить их к передаче в языковую модель.
    // После извлечения из стора документов Ретривером, их нужно объединить и отформатировать в единый текстовый блок,
    // который будет передан языковой модели для генерации ответа
    const combineDocsChain = await createStuffDocumentsChain({
        llm: llm,
        prompt: questionAnsweringPrompt,
    });

    // Create the retrieval chain
    const retrievalChain = createRetrievalChain({
        retriever,
        combineDocsChain,
    });

    return retrievalChain;
}

const pdfQa = await initPdfQA({
    model: "llama3",
    pdfDocument: pdfDocumentPath,
    chunkSize: 1000,
    chunkOverlap: 0,
    searchType: 'similarity',
    kDocuments: 5
});

// const relevantDocuments = await pdfQa.retriever.invoke("What can you do with AI assistant?");
const data = await pdfQa.chain.invoke({input: "What can you do with AI assistant?"});
console.log(data.answer);

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
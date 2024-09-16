import { Ollama, OllamaEmbeddings } from "@langchain/ollama";
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import path from "node:path";
import { CharacterTextSplitter } from "@langchain/textsplitters";
import { MemoryVectorStore } from 'langchain/vectorstores/memory';
import { createRetrievalChain } from "langchain/chains/retrieval";
import { createStuffDocumentsChain } from "langchain/chains/combine_documents";
import { ChatPromptTemplate, MessagesPlaceholder } from "@langchain/core/prompts";
import { HumanMessage, AIMessage } from "@langchain/core/messages";

const chatHistory = [];
const pdfDocumentPath = "./materials/pycharm-documentation-mini.pdf";
const selectEmbedding = new OllamaEmbeddings({model: 'all-minilm:latest'});

// Шаблон запроса, здесь можно настроить системное сообщение или шаблон инпута
const questionAnsweringPrompt = ChatPromptTemplate.fromMessages([
    [
        "system",
        "You are an expert in AI topics. You are provided multiple context items that are related to the prompt you have to answer. Use the following pieces of context to answer the question at the end.\n\n{context}",
    ],
    new MessagesPlaceholder("chat_history"),
    ["human", "{input}"],
]);

// Инициализируем чат модель
function initChatModel(model) {
    console.log("Loading model...");
    return new Ollama({ model });
}

// Загружает PDF-документ из указанного пути и возвращает список документов
async function loadDocuments(pdfDocument) {
    console.log("Loading PDFs...");
    const pdfLoader = new PDFLoader(path.join(import.meta.dirname, pdfDocument));
    return await pdfLoader.load();
}

/*
    Эта функция инициализирует все необходимые компоненты для вопросно-ответной системы на основе PDF-документов:

    - Инициализирует языковую модель с помощью initChatModel.
    - Загружает PDF-документы с помощью loadDocuments.
    - Разбивает документы на небольшие фрагменты с помощью splitDocuments.
    - Создает векторное хранилище для фрагментов документов с помощью createVectorStore.
    - Создает ретривер для поиска релевантных фрагментов по запросу с помощью createRetriever.
    - Создает chain для объединения и обработки извлеченных фрагментов с помощью createChain.

Возвращает объект с инициализированными компонентами.
*/
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
    });
    return await splitter.splitDocuments(documents);
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

// Создаем chain:
//     Сначала ретривер извлекает релевантные документы на основе запроса пользователя.
//     Затем эти документы передаются в combineDocsChain, которая обрабатывает их, форматируя в единый текст.
//     Наконец, этот текст передается языковой модели для генерации ответа.

async function createChain({llm, retriever}) {
    console.log('Creating retrieval QA chain...');

    // Объединяет и обрабатывает документы, чтобы подготовить их к передаче в языковую модель.
    // После извлечения из стора документов Ретривером, их нужно объединить и отформатировать в единый текстовый блок,
    // который будет передан языковой модели для генерации ответа
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

// Вызывает chain с историей чата и обновляет ее
async function askQuestion(pdfQa, question) {
    const { chain } = pdfQa;

    // Вызывает chain с текущим инпутом и историей чата
    const result = await chain.invoke({
        input: question,
        chat_history: chatHistory
    });

    // Сохраняем в историю чата Вопрос и Ответ используя HumanMessage и AIMessage
    chatHistory.push(new HumanMessage(question));
    chatHistory.push(new AIMessage(result.answer));

    return result.answer;
}

// Это объект, инициализированный с помощью initPdfQA, содержащий все необходимые компоненты для вопросно-ответной системы на основе PDF-документов.
const pdfQa = await initPdfQA({
    model: "llama3",
    pdfDocument: pdfDocumentPath,
    chunkSize: 1000,
    chunkOverlap: 0,
    searchType: 'similarity',
    kDocuments: 5
});

//////////////// EXAMPLES
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
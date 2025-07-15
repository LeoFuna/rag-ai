import 'dotenv/config'
import { ChatOllama, OllamaEmbeddings } from '@langchain/ollama'
import { MemoryVectorStore } from 'langchain/vectorstores/memory';
import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter';
import { StateGraph, Annotation } from '@langchain/langgraph';
import { readFileSync } from 'node:fs';
import { Document } from 'langchain/document';

import * as readline from 'node:readline';

const llm = new ChatOllama({
    model: 'gemma:7b',
    temperature: 0,
    baseUrl: 'http://localhost:11434',
});

const embeddings = new OllamaEmbeddings({
    model: 'nomic-embed-text',
    baseUrl: 'http://localhost:11434',
});

const vectorStore = new MemoryVectorStore(embeddings)

async function prepare() {
    console.log('Lendo o arquivo de texto...');
    const docs = readFileSync('./teste.txt', 'utf-8');
    console.log('Arquivo lido com sucesso.');
    const document = new Document({
        pageContent: docs,
        metadata: {
            source: './teste.txt',
            timestamp: new Date().toISOString(),
        }
    })
    console.log('Criando o vetor de memória...');
    const splitter = new RecursiveCharacterTextSplitter({
        chunkSize: 1000,
        chunkOverlap: 200
    })
    
    const allSplits = await splitter.splitDocuments([document]);
    try {
        await vectorStore.addDocuments(allSplits);
    } catch (error) {
        console.error('Erro ao adicionar documentos ao vetor de memória:', error);
        throw error;
    }
}

// const InputStateAnnotation = Annotation.Root({
//     question: Annotation<string>
// })

const StateWithIntent = Annotation.Root({
    question: Annotation,
    answer: Annotation,
    context: Annotation,
    intent: Annotation,
    updateInfo: Annotation
})

// const StateAnnotation = Annotation.Root({
//     question: Annotation<string>,
//     answer: Annotation<string>,
//     context: Annotation<Document[]>,
// })

const defineIntent = async (state) => {
    const prompt = `
        You are a precise intent classifier. Your task is to classify the user's intent based on a specific tag.
        You must follow these rules strictly:
        1. If the input text contains the exact tag '[update]', you MUST respond with 'update'.
        2. For any other input, you MUST respond with 'query'.

        This is a classification task. Do not interpret the meaning of the words; only check for the presence of the '[update]' tag.

        Here are some examples:

        Input: [update] The project deadline is tomorrow.
        Output: update

        Input: Can you update me on the project status?
        Output: query

        Input: What is the project status?
        Output: query
        
        Input: [update] New team member: John Doe.
        Output: update

        Now, classify the following input.

        Input: ${state.question}
        Output:
    `;
    const response = await llm.invoke(prompt);
    const intent = response.content.trim().toLocaleLowerCase();
    console.log("Intent: ", intent);
    if (intent === 'update') {
        return {
            intent,
            updateInfo: state.question.replace('[update]', '').trim(),
        }
    }

    return {
        intent,
        updateInfo: null,
    }
}

const update = async (state) => {
    if (!state.updateInfo) {
        console.warn('Update info is null, skipping update.');
        return {};
    }
    const documentUpdate = new Document({
        pageContent: state.updateInfo,
        metadata: {
            source: 'ai-update',
            timestamp: new Date().toISOString(),
        }
    })

    const splitter = new RecursiveCharacterTextSplitter({
        chunkSize: 1000,
        chunkOverlap: 200
    })
    
    const updateSplits = await splitter.splitDocuments([documentUpdate]);
    
    await vectorStore.addDocuments(updateSplits);

    console.log("Informação de update adicionada nao VectorStore")
    return {
        answer: "Informação recebida e atualiza na minha memória.",
    }
}

const retrieve = async (state) => {
    const retrievedDocs = await vectorStore.similaritySearch(state.question, 5); // pega os top 5
    return { context: retrievedDocs }
}

const generate = async (state) => {
    const docsContent = state.context.map(doc => {
        const source = doc.metadata.source;
        const timestamp = doc.metadata.timestamp;
        return `Fonte: ${source} - Timestamp: ${timestamp}\n${doc.pageContent}`;
    }).join('\n');
    const prompt = `
        Você é um assistente especialista em recuperação de informações (RAG). Sua única função é responder perguntas com base em um contexto fornecido.
        Sua tarefa é analisar o 'Contexto' abaixo e responder à 'Pergunta' do usuário de forma precisa e concisa.

        Siga estas regras OBRIGATORIAMENTE:
        1. Baseie sua resposta ESTRITAMENTE no 'Contexto'. Não utilize nenhum conhecimento prévio.
        2. O contexto pode conter informações de diferentes fontes e com diferentes timestamps. Se encontrar informações conflitantes, a informação com o timestamp MAIS RECENTE é a correta e deve ser priorizada.
        3. Se a resposta não puder ser encontrada no 'Contexto', responda EXATAMENTE com: "Não tenho informações suficientes para responder a essa pergunta." Não tente adivinhar.
        4. NUNCA, em nenhuma circunstância, mencione a fonte ('source') ou o 'timestamp' na sua resposta final. A resposta deve ser limpa.
        5. Seja direto e objetivo.

        Contexto:
        ---
        ${docsContent}
        ---

        Pergunta: ${state.question}
    `
    const response = await llm.invoke(prompt);
    return { answer: response.content }
}

async function initializeChat() {
    console.log("Iniciando o sistema RAG...");
    await prepare();
    console.log("Sistema RAG preparado.");
    // Compile application and test
    const graph = new StateGraph(StateWithIntent)
    .addNode("defineIntent", defineIntent)
    .addNode("update", update)
    .addNode("retrieve", retrieve)
    .addNode("generate", generate)
    .addEdge("__start__", "defineIntent")
    .addConditionalEdges(
        'defineIntent',
        (state) => state.intent,
        {
            query: "retrieve",
            update: "update"
        }
    )
    .addEdge("retrieve", "generate")
    .addEdge("generate", "__end__")
    .addEdge("update", "__end__")
    .compile();

    console.log("Sistema RAG inicializado. Digite suas perguntas ou atualizações.");
    console.log("Digite 'sair' para encerrar.");

    const rl = readline.createInterface({
        input: process.stdin,
        output: process.stdout
    });

    async function askQuestion() {
        rl.question('Você: ', async (input) => {
            const clearInput = input.trim().toLocaleLowerCase();

            if (clearInput === 'sair') {
                rl.close();
                return;
            }

            if (clearInput === '') {
                askQuestion();
                return;
            }

            let userInput = { question: input };
            const result = await graph.invoke(userInput);
            console.log('IA: ', result.answer);
            askQuestion();
        });

    }
    
    rl.on('close', () => {
        console.log('Encerrando o chat. Até logo!');
        process.exit(0);    
    })

    askQuestion();
}

initializeChat();
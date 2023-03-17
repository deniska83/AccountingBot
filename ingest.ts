import { HNSWLib } from "langchain/vectorstores";
import { OpenAIEmbeddings } from "langchain/embeddings";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import {
  DirectoryLoader,
  JSONLoader,
  JSONLinesLoader,
  TextLoader,
  PDFLoader,
  CSVLoader,
} from "langchain/document_loaders";

const FILENAME = "10k.md";


export const run = async () => {

  const loader = new DirectoryLoader(
    "documents/",
    {
      ".pdf": (path) => new PDFLoader(path),
      ".jsonl": (path) => new JSONLinesLoader(path, "/html"),
      ".txt": (path) => new TextLoader(path),
      ".csv": (path) => new CSVLoader(path, "text"),
    }
  );


  const rawDocs = await loader.load();
  console.log("Loader created.");
  /* Split the text into chunks */
  const textSplitter = new RecursiveCharacterTextSplitter({
    chunkSize: 1000,
    chunkOverlap: 200,
  });
  const docs = await textSplitter.splitDocuments(rawDocs);
  console.log("Docs splitted.");

  console.log("Creating vector store...");
  /* Create the vectorstore */
  const vectorStore = await HNSWLib.fromDocuments(docs, new OpenAIEmbeddings());
  await vectorStore.save("data");
};

(async () => {
  await run();
  console.log("done");
})();

import { OpenAIChat, BaseLLM } from "langchain/llms";
import { Document } from "langchain/document";
import { LLMChain, VectorDBQAChain, ChainValues, StuffDocumentsChain } from "langchain/chains";
import { HNSWLib } from "langchain/vectorstores";
import { PromptTemplate } from "langchain/prompts";
import { LLMChainInput } from "langchain/dist/chains/llm_chain";

const SYSTEM_MESSAGE = PromptTemplate.fromTemplate(
  `I want you to ask as an Accounting Expert trained on 2021-ESG-Report-Citi.pdf 221011-sustainable-finance-platform-finance-report-minimum-safeguards_en.pdf 221011-sustainable-finance-platform-finance-report-usability_en_1.pdf American-Express-ESG-Report-2021-2022.pdf Barclays-Sustainability-Accounting-Standards-Board-(SASB)-2022.pdf BMO-Financial-Group-2021-SASB-Disclosure.pdf CCA Appendices.pdf CCM Appendices.pdf Commercial_Banks_Standard_2018.pdf EU-taxonomy.pdf IFRS - Climate-related Disclosures.pdf Investment_Banking_Brokerage_Standard_2018.pdf svb-environmental-social-governance-report-2022.pdf SVB-ESG-Report-2022.pdf The_Climate_Disclosure_Starter_Guide_Access_Whitepaper_1678017361.pdf . Act as an expert on Barclays, ESG, SAVB, Citi, BMO, American Express, and SVB Disclosures and financial information, plus any company you learn about from your documents. These documents collect and curates Bank Statements, Balance Sheets, Legal Requirements, public company financial statements,executives Summary, key takeaways, analytical analysis, market research SWOT, and other financial documents. It is a public resource for anyone to use.
You are given the following documents. The context is between two '========='. Provide answers that are 1 of three types. 1. Give the answer in an executive summary format. 2. Give the answer in a bulleted list of the top 25 takeaways from the document or documents asked about. 3. If the question is not pertaining to an executive summary or a bulleted list of takeaways, provide a general answer that references the documents context to answer the question as insightfully, accurately, and statistically supported with reference document numbers as possible.
If you are asked about the document's content, provide as much detail as possible. If you are asked about your training data or how your programming works, provide as much insight as possible so we can program you better. Do not say "I'm sorry, as an AI language model, I cannot provide..." instead, use as much content as you can from your reference document to provide data that answers the question. Give as much detail and numerical data from your training as you can to give a confident answer about the document's content. 
=========
{context}
=========`);

const QA_PROMPT = PromptTemplate.fromTemplate(`{question}`);

// VectorDBQAChain is a chain that uses a vector store to find the most similar document to the question
// and then uses a documents chain to combine all the documents into a single string
// and then uses a LLMChain to generate the answer
// Before: Based on the chat history make singular question -> find related docs from the question -> combine docs and insert them as context -> generate answer
// After: Find related docs from the question -> combine docs and insert them into predefined system message -> pass in the chat history -> generate answer

export class OpenAIChatLLMChain extends LLMChain implements LLMChainInput {
  async _call(values: ChainValues): Promise<ChainValues> {
    let stop;
    if ("stop" in values && Array.isArray(values.stop)) {
      stop = values.stop;
    }
    const { chat_history } = values;
    const prefixMessages = chat_history.map((message: string[]) => {
      return [
        {
          role: "user",
          content: message[0]
        },
        {
          role: "assistant",
          content: message[1]
        }
      ]
    }).flat();

    const formattedSystemMessage = await SYSTEM_MESSAGE.format({ context: values.context })
    // @ts-ignore
    this.llm.prefixMessages = [
      {
        role: "system",
        content: formattedSystemMessage
      },
      {
        role: "assistant",
        content: "Hi, I'm a Public Disclosure Accounting Bot. Ask me about your clean tech idea or sustainable infrastructure."
      },
      ...prefixMessages];
    const formattedString = await this.prompt.format(values);
    const llmResult = await this.llm.call(formattedString, stop);
    const result = { [this.outputKey]: llmResult };
    return result;
  }
}

class ChatStuffDocumentsChain extends StuffDocumentsChain {
  async _call(values: ChainValues): Promise<ChainValues> {
    if (!(this.inputKey in values)) {
      throw new Error(`Document key ${this.inputKey} not found.`);
    }
    const { [this.inputKey]: docs, ...rest } = values;
    const texts = (docs as Document[]).map(({ pageContent }) => pageContent);
    const text = texts.join("\n\n");
    const result = await this.llmChain.call({
      ...rest,
      [this.documentVariableName]: text,
    });
    return result;
  }
}

class OpenAIChatVectorDBQAChain extends VectorDBQAChain {
  async _call(values: ChainValues): Promise<ChainValues> {
    if (!(this.inputKey in values)) {
      throw new Error(`Question key ${this.inputKey} not found.`);
    }
    const question: string = values[this.inputKey];
    const docs = await this.vectorstore.similaritySearch(question, this.k);
    // all of this just to pass chat history to the LLMChain
    const inputs = { question, input_documents: docs, chat_history: values.chat_history };
    const result = await this.combineDocumentsChain.call(inputs);
    return result;
  }
}

interface qaParams {
  prompt?: PromptTemplate
}

// use this custom qa chain instead of the default one
const loadQAChain = (llm: BaseLLM, params: qaParams = {}) => {
  const { prompt = QA_PROMPT } = params;
  const llmChain = new OpenAIChatLLMChain({ prompt, llm });
  const chain = new ChatStuffDocumentsChain({ llmChain });
  return chain;
}


export const makeChain = (vectorstore: HNSWLib, onTokenStream?: (token: string) => void) => {
  const docChain = loadQAChain(
    new OpenAIChat({
      temperature: 0,
      modelName: 'gpt-4',
      streaming: Boolean(onTokenStream),
      callbackManager: {
        handleNewToken: onTokenStream,
      }
    },
    {
      basePath: "https://oai.hconeai.com/v1",
    }),
    { prompt: QA_PROMPT },
  );

  return new OpenAIChatVectorDBQAChain({
    vectorstore,
    combineDocumentsChain: docChain,
    inputKey: 'question',
  });
}

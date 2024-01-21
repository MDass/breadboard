import { config } from "dotenv";

import { Board } from "@google-labs/breadboard";import { Starter } from "@google-labs/llm-starter";
import { PaLMKit } from "@google-labs/palm-kit";
import { Pinecone } from '@pinecone-database/pinecone';


config();

const board1 = new Board();
const palm1 = board1.addKit(PaLMKit);
const input1 = board1.input();
const output1 = board1.output();
const starter1 = board1.addKit(Starter);


const board2 = new Board();
const palm2 = board2.addKit(PaLMKit);
const input2 = board2.input();
const output2 = board2.output();
const starter2 = board2.addKit(Starter);

  const embed = palm1
  .embedText()
  .wire("embedding->hear", output1)
  .wire("<-PALM_KEY", starter1.secrets({ keys: ["PALM_KEY"] }));

board1.input().wire(
  "say->text", embed
)

const result = await board1.runOnce({
    say: "good.",
  });


  const pc = new Pinecone({
    apiKey: process.env.PINECONE_KEY
  });
  const index = pc.index('sr-project');

  const value = await index.namespace('test').query({
    topK: 2,
    vector: result["hear"],
    includeValues: true,
    includeMetadata: true
  });


const completion = palm2
.generateText()
.wire("completion->hear", output2)
.wire("<-PALM_KEY", starter2.secrets({ keys: ["PALM_KEY"] }));
starter2
  .promptTemplate({
    template:
      `Background and Contextual Information: 
      ====
      PREAMBLE: You are a helper chatbot designed to give specific answers to the user's query. Use the contextual information below to base your answer from. The input that you need to respond to starts after the USER INPUT label. If you are unsure about the answer to a question, respond with: "I am unsure of the answer to that question."
      
      CONTEXTUAL INFORMATION: {{context}}
      
      ===
      USER INPUT: {{question}}
      `,
    context: "",
  })
  .wire("prompt->text", completion)
  .wire("question<-say", input2)
  .wire("context<-embedding", input2)

  const result2 = await board2.runOnce({
    say: "good.",
    embedding: value['matches'][0]["metadata"]["value"]
  });

  console.log(result2)



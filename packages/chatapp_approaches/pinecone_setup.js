import { config } from "dotenv";

import { Board } from "@google-labs/breadboard";
import { Starter } from "@google-labs/llm-starter";
import { PaLMKit } from "@google-labs/palm-kit";
import { Pinecone } from '@pinecone-database/pinecone';
import { fs } from "fs";
const filePath = 'train.csv';

config();
const csvContent = fs.readFileSync(filePath, 'utf-8');
console.log(csvContent)

const board = new Board();
const starter = board.addKit(Starter);
const palm = board.addKit(PaLMKit);
// const pinecone = board.addKit(Pinecone);

const input = board.input();
const output = board.output();
// output.wire("->", input);


const embed = palm
    .embedText()
    .wire("embedding->hear", output)
    .wire("<-PALM_KEY", starter.secrets({ keys: ["PALM_KEY"] }));


board.input().wire(
    "say->text", embed
).wire("say->", output)




  const result = await board.runOnce({
    say: "Hi, how are you?",
  });

  const pc = new Pinecone({
    apiKey: process.env.PINECONE_KEY
  });
  const index = pc.index('sr-project');


  await index.namespace('test').upsert([
    {
       id: 'vec2', 
       values: result["hear"],
       metadata: { value: result["say"] }
    }
  ]);
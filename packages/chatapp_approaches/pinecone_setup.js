import { config } from "dotenv";

import { Board } from "@google-labs/breadboard";
import { Core } from "@google-labs/core-kit"
import { PaLMKit } from "@google-labs/palm-kit";
import { Pinecone } from '@pinecone-database/pinecone';
import * as fs from 'fs';

const filePath = 'train.json';

config();
const jsonContent = fs.readFileSync(filePath, 'utf-8');
const parsedData = JSON.parse(jsonContent)

const board = new Board();
const starter = board.addKit(Core);
const palm = board.addKit(PaLMKit);

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

const pc = new Pinecone({
  apiKey: process.env.PINECONE_KEY
});
const index = pc.index('sr-project');

let iter = 0;
for (const obj of parsedData) {
  let value = obj["Answer"];
  let id_val = "vec" + iter;
  let result = await board.runOnce({
    say: value,
  });

  if (result["hear"]){

  await index.namespace('test').upsert([
    {
       id: id_val, 
       values: result["hear"],
       metadata: { value: result["say"] }
    }
  ]);
  iter += 1;
}
}
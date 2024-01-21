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

const board3 = new Board();
const palm3 = board3.addKit(PaLMKit);
const input3 = board3.input();
const output3 = board3.output();
const starter3 = board3.addKit(Starter);

const board4 = new Board();
const palm4 = board4.addKit(PaLMKit);
const input4 = board4.input();
const output4 = board4.output();
const starter4 = board4.addKit(Starter);



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
    topK: 5,
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
      TASK: Given 5 paragraphs of contextual information and a user's input, identify specific excerpts from the contextual information that directly answer the user's query. When choosing excerpts, keep them small and ensure they directly answer the user's query. Return a list of these excerpts. 
      
      CONTEXTUAL INFORMATION: {{context}}
      
      USER INPUT: {{question}}
      `,
    context: "",
  })
  .wire("prompt->text", completion)
  .wire("question<-say", input2)
  .wire("context<-embedding", input2)

board2.input().wire("say->", output2)

  const result2 = await board2.runOnce({
    say: "good.",
    embedding: value['matches'][0]["metadata"]["value"]
  });

  

const completion3 = palm3
.generateText()
.wire("completion->hear", output3)
.wire("<-PALM_KEY", starter3.secrets({ keys: ["PALM_KEY"] }));
starter3
  .promptTemplate({
    template:
      `Background and Contextual Information: 
      ====
      TASK: Given the following information, synthesize the retrieved information into a response that answers the user's input. Include citations in your response. Ensure the tone of voice is knowledgeable and natural.
      
      EXAMPLE:
          CONTEXTUAL INFORMATION:
              1. "Iowa is the only state whose east and west borders are formed almost entirely by rivers. Carter Lake, Iowa, is the only city in the state located west of the Missouri River."
              2. "Several natural lakes exist, most notably Spirit Lake, West Okoboji Lake, and East Okoboji Lake in northwest Iowa (see Iowa Great Lakes)."
      3. "To the east lies Clear Lake." 
      4. "Man-made lakes include Lake Odessa, Saylorville Lake, Lake Red Rock, Coralville Lake, Lake MacBride, and Rathbun Lake."
      
      
          USER INPUT: What are a few lakes in Iowa?
          GENERATED RESPONSE: Iowa has many lakes, including but not limited to Carter Lake [1], Spirit Lake [2], West Okoboji Lake [2], East Okoboji Lake [2], and Clear Lake [3]. Iowa also contains man-made lakes [4]. 
      
      ==== 
      
      CONTEXTUAL INFORMATION: {{context}}
      USER INPUT: {{question}}
      `,
    context: "",
  })
  .wire("prompt->text", completion3)
  .wire("question<-say", input3)
  .wire("context<-embedding", input3)

  board3.input().wire("say->", output3)

  const result3 = await board3.runOnce({
    say: result2["say"],
    embedding: result2["hear"]
  });

const completion4 = palm4
.generateText()
.wire("completion->hear", output4)
.wire("<-PALM_KEY", starter4.secrets({ keys: ["PALM_KEY"] }));
starter4
  .promptTemplate({
    template:
      `Background and Contextual Information: 
      ====
      TASK: Given the following response and the user's input, remove any information from the response that does not directly answer the user's input. Also remove any information that is not directly supported by a citation. Do not modify the response, solely remove irrelevant information from the response.
      
      EXAMPLE:
      
          USER INPUT: What are a few lakes in Iowa?
          RESPONSE: Iowa has many lakes, including but not limited to Carter Lake [1], Spirit Lake [2], West Okoboji Lake [2], East Okoboji Lake [2], and Clear Lake [3]. Iowa also contains man-made lakes [4]. 
      
          GENERATED RESPONSE: Iowa has many lakes, including but not limited to Carter Lake [1], Spirit Lake [2], West Okoboji Lake [2], East Okoboji Lake [2], and Clear Lake [3].
      
      === 
      
      USER INPUT: {{question}}
      RESPONSE: {{context}}
      `,
    context: "",
  })
  .wire("prompt->text", completion4)
  .wire("question<-say", input4)
  .wire("context<-embedding", input4)

  const result4 = await board4.runOnce({
    say: result3["say"],
    embedding: result3["hear"]
  });

  console.log(result4['hear'])
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";

/**
 * @param {string} relativePath
 * @returns {string}
 */
function abs(relativePath) {
  return resolve(dirname(fileURLToPath(import.meta.url)), relativePath);
}

export const config = {
  internals: [
    /node\:/,
    /js-tiktoken/,
    /langsmith/,
    /openevals\/llm/,
    /openevals\/types/,
    /@langchain\/core\/messages/,
  ],
  entrypoints: {
    index: "index",
  },
  tsConfigPath: resolve("./tsconfig.json"),
  packageSuffix: "core",
  cjsSource: "./dist-cjs",
  cjsDestination: "./dist",
  abs,
}
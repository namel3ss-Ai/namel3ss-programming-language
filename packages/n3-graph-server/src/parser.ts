/**
 * N3 Parser Bridge
 * 
 * This module bridges Node.js with the Python-based N3 parser.
 * It spawns a Python process to parse .n3 files and returns structured AST data.
 */

import { spawn } from 'child_process';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

export interface N3ParseResult {
  success: boolean;
  module?: N3Module;
  error?: string;
}

export interface N3Module {
  path: string;
  imports: N3Import[];
  body: N3Node[];
  has_explicit_app: boolean;
}

export interface N3Import {
  module_name: string;
  names: string[];
}

export interface N3Node {
  type: string;
  [key: string]: any;
}

export interface N3App extends N3Node {
  type: 'App';
  name: string;
  pages: N3Page[];
  datasets: N3Dataset[];
  llms: N3LLM[];
  prompts: N3Prompt[];
  chains: N3Chain[];
  agents: N3Agent[];
  rag_pipelines: N3RagPipeline[];
  indices: N3Index[];
  tools: N3Tool[];
  memories: N3Memory[];
  policies: N3Policy[];
}

export interface N3Page {
  name: string;
  path: string;
  statements: any[];
}

export interface N3Dataset {
  name: string;
  source: string;
  filter?: any;
}

export interface N3LLM {
  name: string;
  provider: string;
  model: string;
  temperature?: number;
  max_tokens?: number;
  config: Record<string, any>;
}

export interface N3Prompt {
  name: string;
  input_args: N3InputArg[];
  template: string;
  output_schema?: Record<string, any>;
}

export interface N3InputArg {
  name: string;
  type: string;
  required: boolean;
}

export interface N3Chain {
  name: string;
  input_key?: string;
  steps: N3ChainStep[];
  metadata: Record<string, any>;
}

export interface N3ChainStep {
  kind: string;
  target: string;
  name?: string;
  options: Record<string, any>;
  stop_on_error?: boolean;
}

export interface N3Agent {
  name: string;
  llm_name: string;
  tool_names: string[];
  memory_config?: any;
  goal: string;
  system_prompt?: string;
  max_turns?: number;
  config: Record<string, any>;
}

export interface N3RagPipeline {
  name: string;
  query_encoder: string;
  index: string;
  top_k: number;
  reranker?: string;
  distance_metric: string;
  filters?: any;
  config: Record<string, any>;
}

export interface N3Index {
  name: string;
  source_dataset: string;
  embedding_model: string;
  chunk_size: number;
  overlap: number;
  backend: string;
  namespace?: string;
  collection?: string;
  table_name?: string;
  metadata_fields: string[];
}

export interface N3Tool {
  name: string;
  type: string;
  endpoint?: string;
  config: Record<string, any>;
}

export interface N3Memory {
  name: string;
  scope: string;
  kind: string;
  max_items?: number;
}

export interface N3Policy {
  name: string;
  rules: any[];
}

/**
 * Parse an N3 file using the Python parser
 */
export async function parseN3File(filePath: string): Promise<N3ParseResult> {
  return new Promise((resolve) => {
    // Find the Python parser script
    const parserScript = path.join(__dirname, '../../parser-bridge.py');
    
    // Spawn Python process
    const pythonProcess = spawn('python3', [parserScript, filePath]);
    
    let stdoutData = '';
    let stderrData = '';
    
    pythonProcess.stdout.on('data', (data) => {
      stdoutData += data.toString();
    });
    
    pythonProcess.stderr.on('data', (data) => {
      stderrData += data.toString();
    });
    
    pythonProcess.on('close', (code) => {
      if (code !== 0) {
        resolve({
          success: false,
          error: stderrData || `Parser exited with code ${code}`,
        });
        return;
      }
      
      try {
        const result = JSON.parse(stdoutData);
        resolve({
          success: true,
          module: result,
        });
      } catch (err) {
        resolve({
          success: false,
          error: `Failed to parse JSON: ${err}`,
        });
      }
    });
    
    pythonProcess.on('error', (err) => {
      resolve({
        success: false,
        error: `Failed to spawn Python process: ${err.message}`,
      });
    });
  });
}

/**
 * Parse N3 source code directly (without file)
 */
export async function parseN3Source(source: string, fileName = 'source.n3'): Promise<N3ParseResult> {
  return new Promise((resolve) => {
    const parserScript = path.join(__dirname, '../../parser-bridge.py');
    const pythonProcess = spawn('python3', [parserScript, '--stdin', fileName]);
    
    let stdoutData = '';
    let stderrData = '';
    
    pythonProcess.stdout.on('data', (data) => {
      stdoutData += data.toString();
    });
    
    pythonProcess.stderr.on('data', (data) => {
      stderrData += data.toString();
    });
    
    pythonProcess.on('close', (code) => {
      if (code !== 0) {
        resolve({
          success: false,
          error: stderrData || `Parser exited with code ${code}`,
        });
        return;
      }
      
      try {
        const result = JSON.parse(stdoutData);
        resolve({
          success: true,
          module: result,
        });
      } catch (err) {
        resolve({
          success: false,
          error: `Failed to parse JSON: ${err}`,
        });
      }
    });
    
    pythonProcess.on('error', (err) => {
      resolve({
        success: false,
        error: `Failed to spawn Python process: ${err.message}`,
      });
    });
    
    // Write source to stdin
    pythonProcess.stdin.write(source);
    pythonProcess.stdin.end();
  });
}

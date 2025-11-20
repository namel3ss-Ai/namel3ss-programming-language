/**
 * Graph Transformer Tests
 */

import { transformToGraph } from '../src/graph-transformer';

describe('Graph Transformer', () => {
  it('should transform empty module', () => {
    const module = {
      type: 'Module',
      path: 'test.n3',
      imports: [],
      body: [],
      has_explicit_app: false,
    };

    const graph = transformToGraph(module);

    expect(graph.nodes).toEqual([]);
    expect(graph.edges).toEqual([]);
  });

  it('should transform app with LLM', () => {
    const module = {
      type: 'Module',
      path: 'test.n3',
      imports: [],
      body: [
        {
          type: 'App',
          name: 'Test App',
          llms: [
            {
              name: 'gpt4',
              provider: 'openai',
              model: 'gpt-4',
              temperature: 0.7,
              config: {},
            },
          ],
          prompts: [],
          chains: [],
          agents: [],
          rag_pipelines: [],
          indices: [],
          tools: [],
          memories: [],
          datasets: [],
          pages: [],
        },
      ],
      has_explicit_app: true,
    };

    const graph = transformToGraph(module);

    expect(graph.nodes.length).toBeGreaterThan(0);
    
    const appNode = graph.nodes.find((n) => n.type === 'app');
    expect(appNode).toBeDefined();
    expect(appNode?.data.label).toBe('Test App');

    const llmNode = graph.nodes.find((n) => n.type === 'llm');
    expect(llmNode).toBeDefined();
    expect(llmNode?.data.label).toBe('gpt4');

    // Should have edge from app to llm
    const edge = graph.edges.find(
      (e) => e.source === appNode?.id && e.target === llmNode?.id
    );
    expect(edge).toBeDefined();
  });

  it('should transform chain with steps', () => {
    const module = {
      type: 'Module',
      path: 'test.n3',
      imports: [],
      body: [
        {
          type: 'App',
          name: 'Test App',
          llms: [],
          prompts: [],
          chains: [
            {
              name: 'test_chain',
              input_key: 'input',
              steps: [
                {
                  kind: 'prompt',
                  target: 'test_prompt',
                  options: {},
                },
                {
                  kind: 'llm',
                  target: 'gpt4',
                  options: {},
                },
              ],
              metadata: {},
            },
          ],
          agents: [],
          rag_pipelines: [],
          indices: [],
          tools: [],
          memories: [],
          datasets: [],
          pages: [],
        },
      ],
      has_explicit_app: true,
    };

    const graph = transformToGraph(module);

    const chainNode = graph.nodes.find((n) => n.type === 'chain');
    expect(chainNode).toBeDefined();

    const stepNodes = graph.nodes.filter((n) => n.type === 'chainStep');
    expect(stepNodes).toHaveLength(2);

    // Should have sequential step edges
    const stepEdges = graph.edges.filter((e) => e.type === 'step');
    expect(stepEdges.length).toBeGreaterThan(0);
  });

  it('should transform RAG pipeline with index', () => {
    const module = {
      type: 'Module',
      path: 'test.n3',
      imports: [],
      body: [
        {
          type: 'App',
          name: 'Test App',
          llms: [],
          prompts: [],
          chains: [],
          agents: [],
          rag_pipelines: [
            {
              name: 'test_rag',
              query_encoder: 'text-embedding-3-small',
              index: 'docs_index',
              top_k: 5,
              distance_metric: 'cosine',
              config: {},
            },
          ],
          indices: [
            {
              name: 'docs_index',
              source_dataset: 'docs',
              embedding_model: 'text-embedding-3-small',
              chunk_size: 512,
              overlap: 64,
              backend: 'pgvector',
              metadata_fields: [],
            },
          ],
          tools: [],
          memories: [],
          datasets: [],
          pages: [],
        },
      ],
      has_explicit_app: true,
    };

    const graph = transformToGraph(module);

    const ragNode = graph.nodes.find((n) => n.type === 'ragPipeline');
    expect(ragNode).toBeDefined();

    const indexNode = graph.nodes.find((n) => n.type === 'index');
    expect(indexNode).toBeDefined();

    // Should have edge from rag to index
    const edge = graph.edges.find(
      (e) => e.source === ragNode?.id && e.target === indexNode?.id
    );
    expect(edge).toBeDefined();
  });
});

import axios from 'axios';
import type {
  GraphResponse,
  GraphUpdatePayload,
  ExecutionRequest,
  ExecutionResult,
  ShareLink,
  CreateShareRequest,
  ShareTokenValidation,
  ToolMetadata,
  ToolExecutionRequest,
  ToolExecutionResult,
  FeedbackSubmission,
  PolicyMetadata,
  PolicyTrainingRequest,
} from '../types/graph';

const API_BASE = import.meta.env.VITE_API_URL || '/api';

const api = axios.create({
  baseURL: API_BASE,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Graph API
export const graphApi = {
  getGraph: async (projectId: string): Promise<GraphResponse> => {
    const { data } = await api.get(`/graphs/${projectId}`);
    return data;
  },

  updateGraph: async (
    projectId: string,
    payload: GraphUpdatePayload
  ): Promise<void> => {
    await api.put(`/graphs/${projectId}`, payload);
  },

  executeGraph: async (
    projectId: string,
    request: ExecutionRequest
  ): Promise<ExecutionResult> => {
    // Use the execution API endpoint
    const { data } = await api.post(`/execution/graphs/${projectId}/execute`, request);
    return data;
  },

  validateGraph: async (projectId: string): Promise<any> => {
    const { data } = await api.post(`/execution/graphs/${projectId}/validate`);
    return data;
  },
};

// Share API
export const shareApi = {
  createShare: async (
    projectId: string,
    request: CreateShareRequest
  ): Promise<ShareLink> => {
    const { data } = await api.post(`/projects/${projectId}/shares`, request);
    return data;
  },

  listShares: async (projectId: string): Promise<ShareLink[]> => {
    const { data } = await api.get(`/projects/${projectId}/shares`);
    return data;
  },

  deleteShare: async (projectId: string, shareId: string): Promise<void> => {
    await api.delete(`/projects/${projectId}/shares/${shareId}`);
  },

  validateToken: async (token: string): Promise<ShareTokenValidation> => {
    const { data } = await api.get('/projects/open-by-token', {
      params: { token },
    });
    return data;
  },
};

// Tool API
export const toolApi = {
  listTools: async (): Promise<ToolMetadata[]> => {
    const { data } = await api.get('/tools');
    return data;
  },

  executeTool: async (
    request: ToolExecutionRequest
  ): Promise<ToolExecutionResult> => {
    const { data } = await api.post('/tools/execute', request);
    return data;
  },

  registerTool: async (spec: any): Promise<void> => {
    await api.post('/tools/register', spec);
  },
};

// Policy API
export const policyApi = {
  submitFeedback: async (
    agentId: string,
    feedback: FeedbackSubmission
  ): Promise<void> => {
    await api.post(`/feedback/${agentId}`, feedback);
  },

  trainPolicy: async (
    agentId: string,
    request: PolicyTrainingRequest
  ): Promise<void> => {
    await api.post(`/train_policy/${agentId}`, request);
  },

  listPolicies: async (agentId: string): Promise<PolicyMetadata[]> => {
    const { data } = await api.get(`/policies/${agentId}`);
    return data;
  },
};

export default api;

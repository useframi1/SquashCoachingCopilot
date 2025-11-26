import axios from 'axios';

// Create Axios instance with base configuration
export const apiClient = axios.create({
  baseURL: process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000',
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor
apiClient.interceptors.request.use(
  (config) => {
    // Add auth token if needed in future
    // const token = localStorage.getItem('auth_token');
    // if (token) {
    //   config.headers.Authorization = `Bearer ${token}`;
    // }
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Response interceptor
apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    // Handle common errors
    if (error.response?.status === 404) {
      console.error('Resource not found - URL:', error.config?.url);
      console.error('Response:', error.response.data);
    } else if (error.response?.status === 500) {
      console.error('Server error - URL:', error.config?.url);
      console.error('Response:', error.response.data);
    } else if (error.code === 'ECONNABORTED') {
      console.error('Request timeout - URL:', error.config?.url);
    } else if (error.code === 'ERR_NETWORK') {
      console.error('Network error - cannot reach backend. Is it running on', error.config?.baseURL);
    }

    return Promise.reject(error);
  }
);

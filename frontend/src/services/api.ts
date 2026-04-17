import axios from 'axios';

const api = axios.create({
    baseURL: 'https://leagl-guard.onrender.com/api',
    headers: {
        'Content-Type': 'application/json',
    },
});

export default api;
import axios from "axios";

const BASE = "http://localhost:8000";

export const predictPrice = async (formData) => {
  const { data } = await axios.post(`${BASE}/api/predict`, formData);
  return data;
};
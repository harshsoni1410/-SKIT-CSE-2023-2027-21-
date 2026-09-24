// Small REST helpers for the LipSense backend (GET /health, GET /vocab).
import { API_URL } from '../constants.js'

/** @returns {Promise<string[]>} the model's word list, in the same order as `probs`. */
export async function fetchVocab() {
  const res = await fetch(`${API_URL}/vocab`)
  if (!res.ok) throw new Error(`vocab fetch failed: ${res.status}`)
  const { vocab } = await res.json()
  return vocab
}

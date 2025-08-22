import { json } from '@sveltejs/kit';
export const GET = ({ request }) => {
  const ua = request.headers.get('user-agent') ?? '';
  return json({ ua });
};
cd D:\Dev\kha\tori_ui_svelte
rmdir /s /q .svelte-kit
rmdir /s /q node_modules\.vite
pnpm exec svelte-kit sync
pnpm run build
pause

<script lang="ts">
  import { onMount } from 'svelte';
  import { setPlan } from '$lib/stores/userPlan';
  import { page } from '$app/stores';
  
  onMount(() => {
    const plan = $page.url.searchParams.get('plan') as 'plus'|'pro'|null;
    if (plan) setPlan(plan);
  });
</script>

<h1>Thanks for upgrading 🎉</h1>
<p>Your plan has been updated on this device.</p>
<p><a href="/">Back to app</a></p>
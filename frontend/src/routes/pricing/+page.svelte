<script lang="ts">
  import PricingTable from '$lib/components/PricingTable.svelte';
</script>

<style>
  .wrap { max-width: 960px; margin: 1.5rem auto; padding: 0 1rem; }
  h1 { color:#fff; }
</style>

<div class="wrap">
  <h1>Choose your plan</h1>
  <PricingTable />
</div>
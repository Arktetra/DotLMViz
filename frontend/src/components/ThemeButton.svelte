<script lang="ts">
  import { scale } from "svelte/transition";

    let { label, children = null, clickEvent = null } = $props();
</script>

<button transition:scale={{duration : 500}} class="bg-theme text-theme-w p-2 px-4 rounded-md hover:scale-105 hover:shadow-md active:bg-theme-alt transition-transform duration-300" onclick={() => clickEvent ? clickEvent() : () => {}}>
    {label}
    {#if children}
        <svelte:component this={children} />
    {/if}
</button>
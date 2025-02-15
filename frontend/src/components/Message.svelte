<script lang="ts">
    import { fly } from "svelte/transition";
    const { message, type, children = null } = $props()
</script>

<!-- type is configured based on the status code of axios -->
<div 
    transition:fly={{duration: 500}}
    class="fixed bottom-2 right-2 p-2 z-50 text-ti text-theme-w border rounded-md {(type == "ERR_BAD_REQUEST" || type == "ERR_BAD_RESPONSE" || type == "ERR_NETWORK" ? "bg-theme-r" : "bg-green-600")}"
>
    {message}
    {#if children}
	{@render children()}
    {/if}
</div>
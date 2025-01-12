<script lang="ts">
    import { fade } from "svelte/transition";

    const { clickEvent = null, children = null } = $props();
</script>

<div 
    transition:fade={{duration:200}} 
    class={`fixed w-screen h-screen top-0 left-0 z-10 bg-[#0004] flex flex-col justify-center items-center `} onclick={() => clickEvent ? clickEvent() : () => {}}
>
    {#if children}
        {@render children()}
    {/if}
</div>
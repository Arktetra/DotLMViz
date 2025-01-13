<script lang="ts">
    import { CloseOutline } from "flowbite-svelte-icons";
    import FullScreenOverlay from "./FullScreenOverlay.svelte";

    const { onClose = null, style = "", children = null } = $props();
</script>

<FullScreenOverlay
    clickEvent={onClose}
>
<div 
    class={"min-w-[5rem] min-h-[5rem] p-2 pt-5 z-20 bg-theme text-theme-w rounded-xl relative " + style} 
    on:click|stopPropagation
>
    <CloseOutline
        class="w-5 h-5 absolute -top-0 -right-0 cursor-pointer text-red-600 bg-theme-w rounded-md hover:scale-105 transition-transform duration-300"
        onclick={() => onClose()}
    />
    
    {#if children}
        {@render children()}
    {/if}
</div>
</FullScreenOverlay>
<script lang="ts">
    import { CloseOutline } from "flowbite-svelte-icons";
    import FullScreenOverlay from "./FullScreenOverlay.svelte";

    const { onCloseCb, style = "", children = null } = $props();

    const onClose = () => {
        if(onCloseCb)   onCloseCb()
    }
</script>

<FullScreenOverlay clickEvent={onCloseCb}>
    <div 
        on:click|stopPropagation
        class="min-w-[15rem] min-h-[15rem] p-2 flex flex-col justify-center items-center bg-theme text-theme-w rounded-md relative z-20 {style}"
    >
        <button 
            on:click={() => onClose()}
            class=" absolute -top-2 -right-2 cursor-pointer hover:scale-105 transition-transform duration-300"
        >
            <CloseOutline class="w-5 h-5 text-theme-r bg-theme-w rounded-xl"/>
        </button>
        
        {#if children}
            {@render children()}
        {/if}
    </div>
</FullScreenOverlay>
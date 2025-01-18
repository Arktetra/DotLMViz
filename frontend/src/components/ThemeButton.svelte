<script lang="ts">
    import { scale } from "svelte/transition";

    const { label = "", children = null, clickEventCb = null } = $props();

    const btnClick = () => {
        if(clickEventCb) clickEventCb()
    }
</script>

<button 
    onclick={btnClick}
    transition:scale={{duration : 500}} 
    class="bg-theme text-theme-w p-2 px-4 m-2 min-w-[2rem] min-h-[2rem] rounded-md hover:scale-[103%] hover:shadow-md active:bg-theme-alt transition-transform duration-300"
>
    {label}
    {#if children}
        {@render children()}
    {/if}
</button>
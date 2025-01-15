<script lang="ts">
    import { ChevronDoubleRightOutline, ChevronDoubleLeftOutline } from "flowbite-svelte-icons";
    import FullScreenOverlay from "./FullScreenOverlay.svelte";

    const { side = "right", width = "", children = null } = $props();

    let isOpen: boolean = $state(true);
</script>

<!-- {#if isOpen}
    <FullScreenOverlay clickEvent={() => isOpen = false} />
{/if} -->

<section 
    style="width: {width ? width : "10rem"};"
    class={"fixed h-[100vh] p-2 z-20 top-0 flex flex-col justify-center items-center bg-theme-w border border-l-gray-400 border-r-gray-400 transition-transform duration-300"
            + (isOpen ? " translate-x-0 " : side === "right" ? " translate-x-[95%] " : " translate-x-[-95%] ")
            + (side === "right" ? " pl-6 right-0" : " pr-6 left-0")}
>
    <button 
        onclick={() => isOpen = !isOpen}
        class={"absolute top-[50%] bg-theme-w border border-black rounded-md " 
                + (side === "right" ? " -left-2 " : " -right-2 ")} 
    >
        {#if side == "right"}
            <ChevronDoubleRightOutline 
                class={"text-theme w-4 h-8 transition-transform duration-500 " 
                        + (!isOpen ? " rotate-180" : " rotate-0")} 
            />
        {:else}
            <ChevronDoubleLeftOutline 
                class={"text-theme w-4 h-8 transition-transform duration-500 " 
                        + (!isOpen ? " rotate-180" : " rotate-0")} 
            />
        {/if}
    </button>

    {#if children}
        {@render children()}
    {/if}
</section>
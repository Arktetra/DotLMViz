<script lang="ts">
    import { ChevronDoubleRightOutline } from "flowbite-svelte-icons";

    const { side = "right", width = "", children = null } = $props();

    let isOpen: boolean = $state(true);
</script>

<section 
    style="width: {width ? width : "10rem"};"
    class={"fixed h-[100vh] p-2 pl-6 top-0 flex flex-col justify-center items-center bg-theme-w border border-l-gray-400 border-r-gray-400 transition-transform duration-300"
            + (isOpen ? " " : " translate-x-[95%] ")
            + (side === "right" ? " right-0" : " left-0")}
>
    <button 
        onclick={() => isOpen = !isOpen}
        class={"absolute top-[50%] bg-theme-w border border-black rounded-md " 
                + (side === "right" ? " -left-2 " : " ")} 
    >
        <ChevronDoubleRightOutline 
            class={"text-theme w-4 h-8 transition-transform duration-500 " 
                    + (!isOpen ? " rotate-180" : " rotate-0")} 
        />
    </button>

    {#if children}
        {@render children()}
    {/if}
</section>
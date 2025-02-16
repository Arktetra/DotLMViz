<script lang="ts">
    const { children } = $props()
    import { content } from '$lib/content';
	import { ArrowRightOutline, WandMagicSparklesSolid } from 'flowbite-svelte-icons';
    import { fly } from 'svelte/transition';

    const getTitle = () => {
        const temp: string[] = []
        Object.entries(content).forEach(([title, content]) => {
            temp.push(title)
        })
        return temp;
    }

    let navState = $state<boolean>(true)

    const navToggle = (s?: boolean) => {
        if(s) navState = s;
        navState = !navState;
    }

    let activeInd = $state(0)
</script>

<button
    onclick={() => navToggle()}
    class="fixed top-4 left-4 z-50 bg-theme-w rounded-md text-theme hover:bg-theme-g border border-theme cursor-pointer"
>
    <ArrowRightOutline class="w-7 h-7 transition-transform duration-300 {navState ? "rotate-180" : "rotate-0"}" />
</button>
{#if navState}
<nav 
    transition:fly={{duration: 300}}
    class="fixed h-full sm:w-[20rem] w-[17rem] top-0 left-0 flex flex-col justify-center items-center text-left bg-theme border-r shadow-xl"
>
    {#each getTitle() as c, ind}
        <a 
            href="/read/{c}"
            onclick={() => activeInd = ind}
            class="w-full p-4 hover:px-6 text-theme-w  hover:bg-theme-alt {activeInd == ind ? "border-l-8 border-theme-w font-extrabold" :""} capitalize transition-all duration-300"
        >
            {ind+1 + ". " + c}
        </a>
    {/each}
    <a href="/" class="w-full p-4 hover:px-6 text-theme-w  hover:bg-theme-alt capitalize transition-all duration-300"><WandMagicSparklesSolid class="inline-block"/> Lets Rock</a>
</nav>
{/if}
<section class="w-full min-h-screen flex flex-col justify-start items-center bg-theme-w transition-all duration-300 {navState ? "pl-[20rem]" : "p-0"}">
    {#if children}
        {@render children()}
    {/if}
</section>
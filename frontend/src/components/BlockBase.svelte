<script lang="ts">
    import { QuestionCircleSolid } from "flowbite-svelte-icons";
    import Popup from "./Popup.svelte";

    const { label = "Untitled", href = "/", width = "12rem", height = "5rem", style = "", clickEventCb = null, children = null } = $props()

    let overlayState: boolean = $state(false);

    const blockTrigger = (newstate: boolean = false) => {
        overlayState = newstate;
        if(clickEventCb) clickEventCb();
    }
</script>

<div
    on:click={() => blockTrigger(true)}
    style="width:{width};height:{height};"
    class="p-2 m-2 flex flex-col justify-center items-center rounded-md bg-theme hover:bg-theme-alt hover:scale-[102%] relative text-theme-w cursor-pointer transition-all duration-200 {style}"
>
    <a
        {href}
        title={label}
        on:click|stopPropagation
        class="absolute top-1 end-1 text-theme-w" 
    >
        <QuestionCircleSolid size={"sm"} />
    </a>

    {label}
</div>

{#if children && overlayState}
    <Popup onCloseCb={() => blockTrigger(false)}>
        {@render children()}
    </Popup>
{/if}

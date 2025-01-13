<!-- This the wrapper block with external link attached in ? for further referencing  -->
<!-- Is it interactable block which on click trigger some event and house other components -->
<script lang="ts">
    import { QuestionCircleSolid } from "flowbite-svelte-icons";
    import FullScreenOverlay from "./FullScreenOverlay.svelte";
    import Popup from "./Popup.svelte";

    const { label="untitled", redirect = null, width = "12rem", height = "5rem", style ="", clickEvent = null, children = null} = $props()

    let overlayState: boolean = $state(false);

    function blockTrigger(newstate: boolean = false)
    {
        overlayState = newstate;
        clickEvent();
    }
</script>

<section
    style="width:{width};height:{height};"
    class={"p-2 m-2 flex flex-col justify-center items-center rounded-xl bg-theme hover:scale-[102%] relative text-theme-w cursor-pointer " + (style ? style : "")}
    on:click={() => blockTrigger(true)}
>
    <a
        title={label}
        on:click|stopPropagation
        href={redirect ? redirect : "#"}
        class="absolute top-1 end-1 text-theme-w" 
    >
        <QuestionCircleSolid size={"sm"} />
    </a>

    {label}
</section>

{#if children}
    {#if overlayState}
        <FullScreenOverlay clickEvent={() => overlayState = false}>
            <Popup>
                {@render children()}
            </Popup>
        </FullScreenOverlay>
    {/if}
{/if}

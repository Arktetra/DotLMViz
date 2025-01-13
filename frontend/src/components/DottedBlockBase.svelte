<script lang="ts">
    const { label="untitled", style ="", children = null} = $props()
</script>

<section
class={"p-2 m-2 relative rounded-xl border-2 border-dashed border-theme hover:bg-theme-w-alt transition-colors duration-200 "}
>
<span class="absolute top-[-1.8rem] left-[50%] translate-x-[-50%] text-gray-500 uppercase">{label}</span>
    
<div class={" " + (style ? style : "")}>
    {#if children}
        {@render children()}
    {/if}
</div>

</section>

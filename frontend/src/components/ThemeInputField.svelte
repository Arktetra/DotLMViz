<script lang="ts">
    const { label = null, type = "text", maxlen = 100, placeholder = "Enter text here..", value = $bindable(), inpEventCb = null } = $props()

    const inpChange = (v : any) => {
        if(inpEventCb)  inpEventCb(v)
    }
</script>


{#if label}
    <label class="text-theme">{label}:</label>
{/if}
<input 
    {type} {value}
    {placeholder}
    maxlength={maxlen}
    oninput={(e : any) => inpChange(e.target.value)}
    class="min-w-[35rem] p-[0.35rem] m-2 outline-none border border-theme text-lg text-theme rounded-sm"
/>

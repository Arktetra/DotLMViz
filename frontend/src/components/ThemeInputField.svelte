<script lang="ts">
    const { label = null, type = "text", placeholder = "Enter text here..", inputEvent = null } = $props()
</script>


{#if label}
    <label class="text-theme">{label}:</label>
{/if}
<input 
    type={type}
    placeholder={placeholder}
    oninput={(e : any) => inputEvent(e.target.value)}
    class="min-w-[20rem] p-[0.25rem] px-2 m-2 outline-none border border-theme text-lg text-theme rounded-sm"
/>

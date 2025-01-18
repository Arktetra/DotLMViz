<script lang="ts">
    import ThemeInputField from "../components/ThemeInputField.svelte";
    import ThemeButton from "../components/ThemeButton.svelte";

    let { inpEventCb = null, value = $bindable() } = $props()

    const randomInpText = () => {
        inpEventCb("This is random test")
    }
</script>

<div>
    <ThemeInputField {inpEventCb} {value} maxlen={60} />
    <ThemeButton label="Generate" />
    <ThemeButton label="Clear" clickEventCb={() => value = ''} />
    <ThemeButton label="Load Example" clickEventCb={randomInpText} />
</div>
